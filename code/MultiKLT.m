classdef MultiKLT < handle
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here

    properties
        % Video path
        path_;

        % Integer number of track object
        object_;

        % Actual length in real world for determing the scalefactor
        %   Integer
        dist_mm_;
        
        % ScaleFactor
        %   Integer
        ScaleFactor_;

        % Instrinsic
        K_;

        % Bboxes M-by-4 matrix of [x y w h] object bounding boxes
        Bboxes_ = [];

        % Bboxes points M-by-4 matrix of [pt1 pt2 pt3 pt4] object bounding boxes
        BboxesPoints_ = [];

        % BoxIds M-by-1 array containing ids associated with each bounding box
        BoxIds_ = [];

        % Points M-by-2 matrix containing tracked points from all objects
        Points_ = [];

        % Points M-by-2 matrix containing previous tracked points from all objects
        oldPoints_ = [];

        % PointIds M-by-1 array containing object id associated with each
        %   point. This array keeps track of which point belongs to which object.
        PointIds_ = [];

        % oldPointIds_ 
        % M-by-1 array containing previous object id associated with each
        %   point. This array keeps track of which point belongs to which bbox.
        oldPointIds_ = [];

        % PointSepIdx_ saving the points seperate index.
        %   M-by-1 array 
        PointSepIdx_ = [];

        % tracker A vision.PointTracker object
        tracker_;
        
        % Displacment of bounding boxes (without scale factor)
        %   frames*n_object*2 array
        bbox2d_ = [];

        % Displacment of bounding boxes (with scale factor)
        %   frames*n_object*2 array
        disp2d_ = [];

        % Time 
        % frames*1 array
        times_=  [];
    end

    methods
        function this = MultiKLT(path, object, length, intrinsics)
            %Constructor
            this.path_ = path;
            this.object_ = object;
            this.dist_mm_ = length;
            this.K_ = intrinsics;
            %             this.roi_ = zeros(n_object,4);
            fprintf("====== Welcom to use MultiObjKLTracker ======\n")
            this.tracker_ = ...
                vision.PointTracker('MaxBidirectionalError',2);
        end

        function drawROI(this, first_frame, n_object)
            %drawROI Summary of this method goes here
            %   drawROI

            fprintf("====== Please draw %d ROI ======\n", n_object)

            marker_img = first_frame; % mark ROI and keypoints on image
            for n_idx = 1:n_object
                % draw ROI
                f1 = figure; imshow(first_frame);
                title('selected ROI and features detected');
                r = drawrectangle;
                % detect fratures
                points = detectMinEigenFeatures(rgb2gray(first_frame) ...
                    , 'ROI', r.Position);
                points = points.Location;
                idx = ones(size(points,1),1)*n_idx;
                this.BoxIds_  = [this.BoxIds_, n_idx];
                this.Points_ = [this.Points_; points];
                this.PointIds_ = [this.PointIds_; idx];
                this.Bboxes_ = [this.Bboxes_; r.Position];
                this.BboxesPoints_ = [this.BboxesPoints_; bbox2points(r.Position)];

                % show roi and features
                marker_img = insertShape(marker_img, 'Rectangle' ...
                    , r.Position, 'LineWidth',2);
                marker_img = insertMarker(marker_img, points ...
                    , '+','Color','white');
                close(f1);
            end
            figure; imshow(marker_img);

            % Update the point tracker
            this.tracker_.initialize(this.Points_, first_frame)

        end

        function track(this, frame)
            % Track the points. Note that some points may be lost.
            [points, isFound] = this.tracker_.step(frame);
            this.Points_ = points(isFound, :);
            this.PointIds_ = this.PointIds_(isFound, :);
            this.oldPoints_ = this.oldPoints_(isFound, :);
            this.oldPointIds_ = this.oldPointIds_(isFound, :);
            % find the index
            for idx = 1:1:size(this.Bboxes_,1)
                lastNumber = find(this.PointIds_==idx);
                lastNumber = lastNumber(end);
                this.PointSepIdx_(idx) = lastNumber;
            end
            newpoints = [];
            newpointsIds = [];

            % To know how many the rest points in the first bbox
            begin = 1;
            last = this.PointSepIdx_(1);
            for i = 1:size(this.Bboxes_,1)

                % Estimate the geometric transformation between the old points
                % and the new points and eliminate outliers
                oldPoints = this.oldPoints_(begin:last, :);
                Points = this.Points_(begin:last, :);
                PointsIds = this.PointIds_(begin:last, :);

                [xform, inlierIdx] = estgeotform2d(oldPoints, Points, 'similarity');
                % The rest points after estgeotform2d
                afterGeotrom2dPts = Points(inlierIdx, :);
                afterGeotrom2dPtsIds = PointsIds(inlierIdx, :);
                this.PointSepIdx_(i) = size(afterGeotrom2dPts,1);

                % Apply the transformation to the bounding box points
                this.BboxesPoints_(4*i-3:4*i,:) = transformPointsForward(xform ...
                    , this.BboxesPoints_(4*i-3:4*i,:));

                % Next bounding box the rest points from which to which
                if i < size(this.Bboxes_,1)
                    begin = last+1;
                    last = this.PointSepIdx_(i+1);
                end
                % save estgeotform2d points and its index
                newpoints = [newpoints; afterGeotrom2dPts];
                newpointsIds = [newpointsIds; afterGeotrom2dPtsIds];
            end
            % save new points into and ids into object
            this.Points_ = newpoints;
            this.PointIds_ = newpointsIds;
        end


        function process(this)
            %run Summary of this method goes here
            %   create video object
            vid = VideoReader(this.path_);

            % draw Line
            frame = readFrame(vid);
            frame = undistortImage(frame,this.K_);
            figure; imshow(frame); title('draw line to determine the ScaleFactor')
            I = drawline;
            dist_px = I.Position(2,1)-I.Position(1,1);
            this.ScaleFactor_ = this.dist_mm_/dist_px;

            %   draw ROI
            frame = readFrame(vid);
            frame = undistortImage(frame,this.K_);
            frame = imgaussfilt(frame,2);
            this.drawROI(frame, this.object_);

            %   read video frame by frame
            % create local variable pts2D
            time = floor(vid.Duration); fps = round(vid.FrameRate);
            N = round(time*fps);
            bbox2d = zeros(N,this.object_,2);

            vid.CurrentTime = 0.0; % start from beginning
            videoPlayer = vision.VideoPlayer(); % for displaying realtime tracking processing
            
            vid_idx = 1;
            while hasFrame(vid)
                frame = readFrame(vid);
                frame = undistortImage(frame,this.K_);
                frame = imgaussfilt(frame,2);
                mark_img = frame; % plot bounding box and marker on image
                this.oldPoints_ = this.Points_;
                this.oldPointIds_ = this.PointIds_;
                this.track(frame);
                fprintf("processing " + num2str(vid_idx) + ", time " + vid.CurrentTime + "sec\n");
                this.times_ = [this.times_,vid.CurrentTime];

                % show the tracking result
                for i = 1:length(this.BoxIds_)
                    bboxPoints = this.BboxesPoints_(4*i-3:4*i,:);
                    % compute the mean coordinate of bbox
                    bbox2d(vid_idx, i,:) = [mean(bboxPoints(:,1)), mean(bboxPoints(:,2))];
                    bboxPolygon = reshape(bboxPoints', 1, []);
                    mark_img = insertShape(mark_img, 'Polygon', bboxPolygon, ...
                        'LineWidth', 2);
                    mark_img = insertMarker(mark_img, this.Points_,'+');
                end
                
                % update the tracker
                setPoints(this.tracker_, this.Points_);
                videoPlayer(mark_img);
                vid_idx = vid_idx + 1;
            end
            this.bbox2d_ = bbox2d;

            % Compute displacment
            for i = 1:length(squeeze(this.bbox2d_(:,1,:)))
                for j = 1:length(this.BoxIds_)
                    initXY = this.bbox2d_(1,j,:);
                    this.disp2d_(i,j,:) = this.bbox2d_(i,j,:)-initXY;
                end
            end
            this.disp2d_ = this.ScaleFactor_*this.disp2d_;
        end

        function save(this)
            for i = 1:length(this.BoxIds_)
                saveResult = zeros(size(this.disp2d_,1),3);
                saveResult(:,1) = this.times_;
                saveResult(:,2:3) = squeeze(this.disp2d_(:,i,:));
                saveResult = array2table(saveResult, 'VariableNames' ...
                    , {'time', 'u_displacement(mm)', 'v_displacement(mm)'});
                writetable(saveResult,['point_' num2str(i) '_klt.csv'])
            end
        end
    end
end
